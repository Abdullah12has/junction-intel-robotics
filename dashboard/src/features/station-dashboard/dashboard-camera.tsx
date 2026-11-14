import { useEffect, useState } from 'react';

import { SchemaSessionCamera } from '../../api/openapi-spec';
import { CameraView } from '../cameras/camera-view';
import { useRuntimeSession } from '../runtime-session/runtime-session-provider';

import classes from './station-dashboard.module.css';

export const DashboardCamera = ({ camera }: { camera: SchemaSessionCamera }) => {
    const { frame } = useRuntimeSession();
    const source = frame(camera.key);
    const [received, setReceived] = useState<number | null>(null);
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        const unsubscribe = source.subscribe(() => {
            const time = Date.now();
            setReceived(time);
            setNow(time);
        });
        const timer = setInterval(() => setNow(Date.now()), 500);
        return () => {
            unsubscribe();
            clearInterval(timer);
        };
    }, [source]);
    const fresh = received !== null && now - received < 2500;
    return (
        <section className={classes.camera}>
            <div className={classes.cameraHeading}>
                <h3>{camera.name}</h3>
                <span>{fresh ? 'Live' : received === null ? 'Waiting for frames' : 'Stale feed'}</span>
            </div>
            <div className={classes.cameraImage}>
                <CameraView
                    source={source}
                    label={camera.name}
                    width={camera.width}
                    height={camera.height}
                    warning={received !== null && !fresh ? 'No recent frames — image is stale' : null}
                />
            </div>
        </section>
    );
};
