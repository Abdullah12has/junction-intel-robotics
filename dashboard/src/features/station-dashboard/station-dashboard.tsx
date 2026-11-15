import { useState } from 'react';

import { Button, Checkbox, Item, Picker, Switch } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { paths } from '../../paths';
import { useProjectId } from '../projects/use-project';
import { RuntimeSessionProvider, useRuntimeSession } from '../runtime-session/runtime-session-provider';
import { DashboardCamera } from './dashboard-camera';
import { JointTable } from './joint-table';
import { JoystickControls } from './joystick-controls';
import { useTelemetry } from './use-telemetry';

import classes from './station-dashboard.module.css';

const DashboardControls = () => {
    const { project_id } = useProjectId();
    const params = { params: { path: { project_id } } };
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', params);
    const { data: robots } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', params);
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', params);
    const { state, error, statuses, send, isLoading } = useRuntimeSession();
    const { observation, age, fresh } = useTelemetry();
    const [environmentId, setEnvironmentId] = useState(environments[0]?.id ?? '');
    const [cameraIds, setCameraIds] = useState<string[]>(() =>
        cameras.flatMap((camera) => (camera.id ? [camera.id] : []))
    );
    const [pending, setPending] = useState(false);
    const [manual, setManual] = useState(true);
    const [message, setMessage] = useState<string | null>(null);
    const environment = environments.find((item) => item.id === environmentId);
    const connected = state?.environment != null && error?.session_alive !== false;
    const following = connected && state?.control?.type === 'teleop';
    const busy = pending || isLoading;
    const failed = Object.values(statuses).filter((status) => status.state === 'failed');
    const hasLeader = state?.environment?.leader_id != null;
    const nameOf = (id: string) => robots.find((robot) => robot.id === id)?.name ?? id;
    const run = async (command: Parameters<typeof send>[0]) => {
        setPending(true);
        setMessage(null);
        try {
            const ack = await send(command);
            if (!ack.ok) setMessage(ack.error?.message ?? 'The command was refused.');
        } catch (cause) {
            setMessage(cause instanceof Error ? cause.message : 'Command failed.');
        } finally {
            setPending(false);
        }
    };
    const connect = () => {
        if (!environment) return;
        void run({
            command: 'load_environment',
            unload_incompatible: false,
            environment: {
                id: environment.id,
                name: environment.name,
                robots: (environment.robots ?? []).map((robot) =>
                    manual ? { ...robot, tele_operator: { type: 'none' as const } } : robot
                ),
                cameras: cameraIds.map((camera_id) => ({ camera_id })),
            },
        });
    };
    return (
        <main className={classes.dashboard}>
            <header className={classes.header}>
                <div>
                    <p className={classes.eyebrow}>Physical AI Studio · station controls</p>
                    <h1>Robot dashboard</h1>
                    <p>Arms, camera feeds and joint inputs in one place.</p>
                </div>
                <div className={classes.connection} role='status'>
                    <span className={classes.indicator} data-live={connected && fresh} />
                    {state === null
                        ? 'Server disconnected'
                        : connected
                          ? fresh
                              ? following
                                  ? 'Following leader'
                                  : 'Connected · holding'
                              : 'Waiting for fresh telemetry'
                          : 'This dashboard is disconnected'}
                </div>
            </header>
            {(message || error || failed.length > 0) && (
                <div role='alert' className={classes.alert}>
                    {message ??
                        error?.message ??
                        failed.map((item) => item.error ?? `${item.component} failed`).join('; ')}
                    <p>Another Studio tab may own the devices. Close its session before connecting here.</p>
                </div>
            )}
            <div className={classes.layout}>
                <aside className={classes.panel} aria-label='Station controls'>
                    <h2>Arm connection</h2>
                    <p className={classes.hint}>
                        Only one session can own the arms. If another Studio tab is connected, stop its session using
                        the runtime-session button in the footer before connecting here.
                    </p>
                    <Picker
                        label='Environment'
                        selectedKey={environmentId}
                        items={environments}
                        width='100%'
                        isDisabled={connected || busy}
                        onSelectionChange={(key) => setEnvironmentId(String(key ?? ''))}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>
                    <Checkbox isSelected={manual} isDisabled={connected || busy} onChange={setManual}>
                        Manual control · follower only
                    </Checkbox>
                    <div className={classes.devices}>
                        {environment?.robots?.map((robot) => (
                            <div key={robot.robot_id}>
                                <p>
                                    <span>Follower</span>
                                    <strong>{nameOf(robot.robot_id)}</strong>
                                </p>
                                <p>
                                    <span>Leader input</span>
                                    <strong>
                                        {!manual && robot.tele_operator.type === 'robot'
                                            ? nameOf(robot.tele_operator.robot_id)
                                            : 'Not configured'}
                                    </strong>
                                </p>
                            </div>
                        ))}
                    </div>
                    <Button
                        variant={connected ? 'secondary' : 'accent'}
                        width='100%'
                        isDisabled={state === null || busy || !environment}
                        onPress={() => (connected ? void run({ command: 'unload_environment' }) : connect())}
                    >
                        {busy
                            ? 'Applying…'
                            : connected
                              ? 'Disconnect arms & cameras'
                              : 'Connect arms & selected cameras'}
                    </Button>
                    <p className={classes.hint}>
                        Connecting enables follower holding torque. Disconnecting releases the session; it is not a
                        power switch.
                    </p>
                    <div className={classes.control}>
                        <Switch
                            isEmphasized
                            isSelected={following}
                            isDisabled={
                                !connected || busy || (!following && (!fresh || !hasLeader || failed.length > 0))
                            }
                            onChange={(enabled) =>
                                void run({
                                    command: 'set_control',
                                    control: enabled ? { type: 'teleop', hz: 100, grace_s: 3 } : null,
                                })
                            }
                        >
                            Follow leader
                        </Switch>
                        <Button
                            variant='negative'
                            width='100%'
                            isDisabled={!connected}
                            onPress={() => void run({ command: 'set_control', control: null })}
                        >
                            Stop following
                        </Button>
                        <p className={classes.hint}>
                            Stop holds the current pose. Physical power disconnect remains the emergency stop.
                        </p>
                    </div>
                    <h2>Camera selection</h2>
                    <p className={classes.hint}>
                        All cameras are selected by default. Choose feeds before connecting; selection applies to this
                        session.
                    </p>
                    <div className={classes.choices}>
                        {cameras.map((camera) => {
                            const cameraId = camera.id;
                            if (!cameraId) return null;
                            return (
                                <Checkbox
                                    key={camera.id}
                                    isSelected={cameraIds.includes(cameraId)}
                                    isDisabled={connected || busy}
                                    onChange={(checked) =>
                                        setCameraIds((ids) =>
                                            checked ? [...ids, cameraId] : ids.filter((id) => id !== cameraId)
                                        )
                                    }
                                >
                                    {camera.name}
                                </Checkbox>
                            );
                        })}
                        {cameras.length === 0 && <p>No cameras configured.</p>}
                    </div>
                    <a href={paths.project.environments.index({ project_id })}>Open full Studio configuration</a>
                </aside>
                <div className={classes.content}>
                    <JoystickControls
                        connected={connected}
                        ready={connected && fresh && !busy && !state?.control && failed.length === 0}
                        observation={observation}
                        send={send}
                    />
                    <section aria-label='Camera feeds' className={classes.feeds}>
                        {connected &&
                            state?.environment?.cameras.map((camera) => (
                                <DashboardCamera key={camera.camera_id} camera={camera} />
                            ))}
                        {(!connected || state?.environment?.cameras.length === 0) && (
                            <div className={classes.empty}>
                                <h2>Camera feeds</h2>
                                <p>Select cameras and connect to see live views here.</p>
                                <p>Preview is limited to 2 updates/sec to keep the remote connection responsive.</p>
                            </div>
                        )}
                    </section>
                    <section className={classes.panel}>
                        <div className={classes.sectionHeading}>
                            <h2>Joint inputs & feedback</h2>
                            <span>
                                {fresh
                                    ? 'Receiving live samples'
                                    : age === null
                                      ? 'No samples yet'
                                      : `Last sample ${(age / 1000).toFixed(1)}s ago`}
                            </span>
                        </div>
                        <p className={classes.hint}>
                            Measured = follower position. Command = leader-derived target while following. Values use
                            the arm’s configured units; unavailable or stale values show —.
                        </p>
                        <JointTable
                            names={state?.environment?.joint_names ?? []}
                            observation={connected ? observation : undefined}
                            following={following}
                        />
                    </section>
                    <section className={classes.panel}>
                        <h2>Device status</h2>
                        {Object.values(statuses).length === 0 ? (
                            <p className={classes.hint}>Status appears after connecting.</p>
                        ) : (
                            <ul className={classes.statuses}>
                                {Object.values(statuses).map((status) => (
                                    <li key={status.component}>
                                        <span>{status.component}</span>
                                        <strong>{status.state}</strong>
                                        {status.error && <p>{status.error}</p>}
                                    </li>
                                ))}
                            </ul>
                        )}
                    </section>
                </div>
            </div>
        </main>
    );
};

export const StationDashboard = () => {
    const { project_id } = useProjectId();
    // Manual loading avoids re-enabling hardware automatically after a reconnect.
    return (
        <RuntimeSessionProvider project_id={project_id} environment={null} hz={2}>
            <DashboardControls />
        </RuntimeSessionProvider>
    );
};
