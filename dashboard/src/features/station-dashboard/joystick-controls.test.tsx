import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';

import { render } from '../../test-utils/render';
import { JoystickControls } from './joystick-controls';

const observation = {
    type: 'observation' as const,
    timestamp: 10,
    state: { 'wrist_roll.pos': 0 },
    action: null,
    frames: [],
};

describe('manual joystick', () => {
    it('requires explicit enabling and sends a bounded jog only while pressed', async () => {
        const send = rs.fn().mockResolvedValue({ type: 'ack', request_id: '1', ok: true });
        render(<JoystickControls connected ready observation={observation} send={send} />);
        const button = screen.getByRole('button', { name: 'Wrist rotation increase' });
        expect(button).toBeDisabled();
        expect(send).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole('checkbox'));
        fireEvent.keyDown(button, { key: 'Enter' });
        expect(send).toHaveBeenCalledWith({
            command: 'jog_joint',
            joint: 'wrist_roll.pos',
            delta: 0.5,
            observed_at: 10,
        });
        fireEvent.keyUp(button, { key: 'Enter' });
        expect(send).toHaveBeenLastCalledWith({ command: 'set_control', control: null });
    });
    it('does not enable with stale telemetry, but always allows stop when connected', () => {
        const send = rs.fn().mockResolvedValue({ ok: true });
        render(<JoystickControls connected ready={false} observation={undefined} send={send} />);
        expect(screen.getByRole('checkbox')).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Stop movement · Esc' }));
        expect(send).toHaveBeenCalledWith({ command: 'set_control', control: null });
    });
    it('disarms and stops on focus loss', () => {
        const send = rs.fn().mockResolvedValue({ ok: true });
        render(<JoystickControls connected ready observation={observation} send={send} />);
        fireEvent.click(screen.getByRole('checkbox'));
        fireEvent.keyDown(screen.getByRole('button', { name: 'Base increase' }), { key: 'Enter' });
        fireEvent.blur(window);
        expect(screen.getByRole('checkbox')).not.toBeChecked();
        expect(send).toHaveBeenLastCalledWith({ command: 'set_control', control: null });
    });
});
