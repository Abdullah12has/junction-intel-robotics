import { SchemaObservationMessage } from '../../api/openapi-spec';

import classes from './station-dashboard.module.css';

const format = (value: number | undefined) => (value !== undefined && Number.isFinite(value) ? value.toFixed(1) : '—');

export const JointTable = ({
    names,
    observation,
    following,
}: {
    names: string[];
    observation?: SchemaObservationMessage;
    following: boolean;
}) => (
    <div className={classes.tableScroll}>
        <table className={classes.table}>
            <caption>Joint feedback and teleoperation commands</caption>
            <thead>
                <tr>
                    <th scope='col'>Joint</th>
                    <th scope='col'>Measured</th>
                    <th scope='col'>Command</th>
                    <th scope='col'>Difference</th>
                </tr>
            </thead>
            <tbody>
                {names.map((name) => {
                    const measured = observation?.state?.[name];
                    const command = following ? observation?.action?.[name] : undefined;
                    const difference = measured !== undefined && command !== undefined ? command - measured : undefined;
                    return (
                        <tr key={name}>
                            <th scope='row'>{name.replace(/\.pos$/, '').replaceAll('_', ' ')}</th>
                            <td>{format(measured)}</td>
                            <td>{format(command)}</td>
                            <td>{format(difference)}</td>
                        </tr>
                    );
                })}
                {names.length === 0 && (
                    <tr>
                        <td colSpan={4}>Connect the arms to receive joint inputs.</td>
                    </tr>
                )}
            </tbody>
        </table>
    </div>
);
