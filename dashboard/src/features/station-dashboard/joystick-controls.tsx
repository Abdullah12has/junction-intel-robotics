import { useCallback, useEffect, useRef, useState } from 'react';

import { SchemaAckMessage, SchemaObservationMessage } from '../../api/openapi-spec';
import { useRuntimeSession } from '../runtime-session/runtime-session-provider';

import classes from './joystick-controls.module.css';

interface Props {
    connected: boolean;
    ready: boolean;
    observation: SchemaObservationMessage | undefined;
    send: ReturnType<typeof useRuntimeSession>['send'];
}

const pads = [
    { title: 'Base & shoulder', x: 'shoulder_pan', y: 'shoulder_lift', xLabel: 'Base', yLabel: 'Shoulder' },
    { title: 'Elbow & wrist', x: 'elbow_flex', y: 'wrist_flex', xLabel: 'Elbow', yLabel: 'Wrist bend' },
    { title: 'Wrist & gripper', x: 'wrist_roll', y: 'gripper', xLabel: 'Wrist rotation', yLabel: 'Gripper' },
];

export const JoystickControls = ({ connected, ready, observation, send }: Props) => {
    const [armed, setArmed] = useState(false);
    const [active, setActive] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const latest = useRef({ ready, observation, send, armed });
    const generation = useRef(0);
    const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const pending = useRef(false);
    const engaged = useRef(false);

    useEffect(() => {
        latest.current = { ready, observation, send, armed };
    }, [ready, observation, send, armed]);

    const stop = useCallback(() => {
        generation.current += 1;
        if (timer.current !== null) clearTimeout(timer.current);
        setActive(null);
        if (engaged.current) {
            engaged.current = false;
            void latest.current.send({ command: 'set_control', control: null });
        }
    }, []);

    useEffect(() => {
        if (!ready) {
            stop();
            setArmed(false);
        }
    }, [ready, stop]);

    useEffect(() => {
        const leave = () => {
            stop();
            setArmed(false);
        };
        const hide = () => {
            if (document.hidden) leave();
        };
        const escape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') leave();
        };
        window.addEventListener('blur', leave);
        document.addEventListener('visibilitychange', hide);
        window.addEventListener('keydown', escape);
        return () => {
            window.removeEventListener('blur', leave);
            document.removeEventListener('visibilitychange', hide);
            window.removeEventListener('keydown', escape);
            stop();
        };
    }, [stop]);

    const start = (joint: string, direction: number) => {
        if (!latest.current.ready || !latest.current.armed || engaged.current || pending.current) return;
        const token = ++generation.current;
        engaged.current = true;
        setActive(`${joint}:${direction}`);
        setError(null);
        const tick = async () => {
            const current = latest.current;
            if (token !== generation.current || !current.ready || !current.armed || !current.observation) return;
            pending.current = true;
            let ack: SchemaAckMessage;
            try {
                ack = await current.send({
                    command: 'jog_joint',
                    joint: `${joint}.pos`,
                    delta: direction * 0.5,
                    observed_at: current.observation.timestamp,
                });
            } catch (cause) {
                ack = {
                    type: 'ack',
                    request_id: '',
                    ok: false,
                    error: { error_code: 'jog_failed', message: cause instanceof Error ? cause.message : 'Jog failed' },
                };
            } finally {
                pending.current = false;
            }
            if (token !== generation.current) return;
            if (!ack.ok) {
                stop();
                setArmed(false);
                setError(ack.error?.message ?? 'Movement refused.');
                return;
            }
            timer.current = setTimeout(() => void tick(), 400);
        };
        void tick();
    };

    const directionButton = (joint: string, direction: number, label: string, arrow: string, position: string) => (
        <button
            type='button'
            className={`${classes.direction} ${classes[position]}`}
            aria-label={`${label} ${direction > 0 ? 'increase' : 'decrease'}`}
            disabled={!ready || !armed}
            data-active={active === `${joint}:${direction}`}
            onPointerDown={(event) => {
                if (event.button !== 0) return;
                event.preventDefault();
                event.currentTarget.setPointerCapture(event.pointerId);
                start(joint, direction);
            }}
            onPointerUp={stop}
            onPointerCancel={stop}
            onLostPointerCapture={stop}
            onKeyDown={(event) => {
                if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
                    event.preventDefault();
                    start(joint, direction);
                }
            }}
            onKeyUp={(event) => {
                if (event.key === ' ' || event.key === 'Enter') {
                    event.preventDefault();
                    stop();
                }
            }}
            onBlur={stop}
        >
            <span aria-hidden='true'>{arrow}</span>
        </button>
    );

    return (
        <section className={classes.panel} aria-label='Joystick controls'>
            <header className={classes.header}>
                <div>
                    <p className={classes.eyebrow}>MANUAL CONTROL</p>
                    <h2>Jog the follower</h2>
                </div>
                <span className={classes.badge}>{active ? 'Moving' : armed && ready ? 'Ready' : 'Locked'}</span>
            </header>
            <p className={classes.hint}>
                Hold a direction to move slowly. Release to hold position. Each step is limited to 0.5 calibrated units.
                These controls move individual joints, not the gripper in Cartesian space.
            </p>
            <label className={classes.arm}>
                <input
                    type='checkbox'
                    checked={armed && ready}
                    disabled={!ready}
                    onChange={(event) => {
                        stop();
                        setArmed(event.target.checked);
                    }}
                />
                Enable joystick · workspace is clear
            </label>
            {!ready && (
                <p className={classes.hint}>
                    Connect in manual mode and wait for fresh feedback. Stop following or model control before enabling
                    the joystick.
                </p>
            )}
            <div className={classes.pads}>
                {pads.map((pad) => (
                    <div className={classes.group} key={pad.title}>
                        <h3>{pad.title}</h3>
                        <div className={classes.pad}>
                            {directionButton(pad.y, 1, pad.yLabel, '↑', 'up')}
                            {directionButton(pad.x, -1, pad.xLabel, '←', 'left')}
                            <span className={classes.center} aria-hidden='true'>
                                ●
                            </span>
                            {directionButton(pad.x, 1, pad.xLabel, '→', 'right')}
                            {directionButton(pad.y, -1, pad.yLabel, '↓', 'down')}
                        </div>
                        <p className={classes.legend}>
                            ↔ {pad.xLabel}
                            <br />↕ {pad.yLabel}
                        </p>
                    </div>
                ))}
            </div>
            {error && (
                <p role='alert' className={classes.error}>
                    {error}
                </p>
            )}
            <button
                type='button'
                className={classes.stop}
                disabled={!connected}
                onClick={() => {
                    stop();
                    setArmed(false);
                    void send({ command: 'set_control', control: null });
                }}
            >
                Stop movement · Esc
            </button>
            <p className={classes.hint}>
                Stop holds position; it does not remove power. A lost connection allows at most the current small step
                to finish. Use the physical power disconnect in an emergency.
            </p>
        </section>
    );
};
