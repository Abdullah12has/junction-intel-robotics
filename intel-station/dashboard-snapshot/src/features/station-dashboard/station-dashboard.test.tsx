import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { StationDashboard } from './station-dashboard';

const fixture = rs.hoisted(() => ({
    send: rs.fn(),
    connected: false,
    fresh: false,
    following: false,
}));
rs.mock('../runtime-session/runtime-session-provider', () => ({
    RuntimeSessionProvider: ({ children }: { children: React.ReactNode }) => children,
    useRuntimeSession: () => ({
        state: {
            session_id: 's',
            environment: fixture.connected ? { leader_id: 'leader', joint_names: [], cameras: [] } : null,
            control: fixture.following ? { type: 'teleop' } : null,
        },
        error: null,
        statuses: {},
        send: fixture.send,
        isLoading: false,
    }),
}));
rs.mock('./use-telemetry', () => ({
    useTelemetry: () => ({ fresh: fixture.fresh, age: null, observation: undefined }),
}));

const show = () =>
    render(<StationDashboard />, { route: '/projects/p/dashboard', path: '/projects/:project_id/dashboard' });

describe('station controls', () => {
    beforeEach(() => {
        fixture.connected = false;
        fixture.fresh = false;
        fixture.following = false;
        fixture.send.mockReset().mockResolvedValue({ ok: true });
        server.use(
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([{ id: 'env', name: 'Bench', robots: [], cameras: [] }])
            ),
            http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/cameras', () => HttpResponse.json([]))
        );
    });
    it('does not connect or start motion just by opening the dashboard', async () => {
        show();
        expect(await screen.findByRole('button', { name: 'Connect arms & selected cameras' })).toBeEnabled();
        expect(fixture.send).not.toHaveBeenCalled();
        expect(screen.getByRole('switch', { name: 'Follow leader' })).toBeDisabled();
    });
    it('blocks starting with stale feedback but keeps the stop control available', async () => {
        fixture.connected = true;
        show();
        const user = userEvent.setup();
        expect(await screen.findByRole('switch', { name: 'Follow leader' })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Stop following' }));
        expect(fixture.send).toHaveBeenCalledWith({ command: 'set_control', control: null });
    });
    it('sends teleoperation only after an explicit switch action with fresh feedback', async () => {
        fixture.connected = true;
        fixture.fresh = true;
        show();
        const user = userEvent.setup();
        await user.click(await screen.findByRole('switch', { name: 'Follow leader' }));
        expect(fixture.send).toHaveBeenCalledWith({
            command: 'set_control',
            control: { type: 'teleop', hz: 100, grace_s: 3 },
        });
    });
});
