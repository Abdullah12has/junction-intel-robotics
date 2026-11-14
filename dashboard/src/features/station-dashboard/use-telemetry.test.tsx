import { rs } from '@rstest/core';
import { act } from '@testing-library/react';

import { SchemaObservationMessage } from '../../api/openapi-spec';
import { renderHook } from '../../test-utils/render';
import { useTelemetry } from './use-telemetry';

const fixture = rs.hoisted(() => ({ listener: undefined as ((sample: SchemaObservationMessage) => void) | undefined }));
rs.mock('../runtime-session/runtime-session-provider', () => ({
    useRuntimeSession: () => ({ observations, state: { environment: {} } }),
}));
const observations = {
    subscribe: (listener: (sample: SchemaObservationMessage) => void) => {
        fixture.listener = listener;
        return () => {
            fixture.listener = undefined;
        };
    },
};

describe('telemetry freshness', () => {
    afterEach(() => {
        rs.useRealTimers();
    });
    it('expires live data after two seconds without another sample', () => {
        rs.useFakeTimers();
        const { result, unmount } = renderHook(() => useTelemetry());
        expect(result.current.fresh).toBe(false);
        act(() =>
            fixture.listener?.({
                type: 'observation',
                timestamp: 1,
                frames: [],
                state: { 'gripper.pos': 5 },
                action: null,
            })
        );
        expect(result.current.fresh).toBe(true);
        act(() => rs.advanceTimersByTime(2500));
        expect(result.current.fresh).toBe(false);
        expect(result.current.observation).toBeUndefined();
        unmount();
        expect(fixture.listener).toBeUndefined();
    });
});
