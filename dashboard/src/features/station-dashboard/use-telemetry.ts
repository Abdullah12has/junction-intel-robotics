import { useEffect, useState } from 'react';

import { SchemaObservationMessage } from '../../api/openapi-spec';
import { useRuntimeSession } from '../runtime-session/runtime-session-provider';

/** Receipt age is local: the laptop's clock can differ from the browser's. */
export const useTelemetry = () => {
    const { observations, state } = useRuntimeSession();
    const [sample, setSample] = useState<{ observation: SchemaObservationMessage; received: number } | null>(null);
    const [now, setNow] = useState(Date.now);

    useEffect(() => {
        const unsubscribe = observations.subscribe((observation) => {
            const received = Date.now();
            setSample({ observation, received });
            setNow(received);
        });
        const timer = setInterval(() => setNow(Date.now()), 500);
        return () => {
            unsubscribe();
            clearInterval(timer);
        };
    }, [observations]);

    const age = sample === null ? null : Math.max(0, now - sample.received);
    const fresh = state?.environment != null && age !== null && age < 2000;
    return { observation: fresh ? sample?.observation : undefined, age, fresh };
};
