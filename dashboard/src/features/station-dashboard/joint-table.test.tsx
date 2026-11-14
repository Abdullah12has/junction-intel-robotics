import { screen, within } from '@testing-library/react';

import { render } from '../../test-utils/render';
import { JointTable } from './joint-table';

const observation = {
    type: 'observation' as const,
    timestamp: 1,
    frames: [],
    state: { 'wrist_roll.pos': 10 },
    action: { 'wrist_roll.pos': 15 },
};

describe('JointTable', () => {
    it('shows measured feedback, command and their difference', () => {
        render(<JointTable names={['wrist_roll.pos']} observation={observation} following />);
        const row = screen.getByRole('row', { name: /wrist roll/ });
        expect(within(row).getByText('10.0')).toBeInTheDocument();
        expect(within(row).getByText('15.0')).toBeInTheDocument();
        expect(within(row).getByText('5.0')).toBeInTheDocument();
    });
    it('does not display an old command when following is stopped', () => {
        render(<JointTable names={['wrist_roll.pos']} observation={observation} following={false} />);
        expect(screen.queryByText('15.0')).not.toBeInTheDocument();
        expect(screen.getAllByText('—')).toHaveLength(2);
    });
    it('shows unavailable values instead of fabricating zero feedback', () => {
        render(<JointTable names={['wrist_roll.pos']} following={false} />);
        expect(screen.getAllByText('—')).toHaveLength(3);
        expect(screen.queryByText('0.0')).not.toBeInTheDocument();
    });
});
