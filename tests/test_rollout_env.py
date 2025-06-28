import pytest
import numpy as np

from nibabel.streamlines.array_sequence import ArraySequence
from tractoracle_irt.environments.rollout_env import RolloutEnvironment

@pytest.fixture
def get_fake_streamlines():
    nb_streamlines = 100
    rng = np.random.RandomState(42)
    streamlines = rng.rand(nb_streamlines, 256, 3).astype(np.float32)
    lengths = rng.randint(1, 256, size=nb_streamlines).astype(np.int32)
    return streamlines, lengths



def test__padded_streamlines_to_array_sequence(get_fake_streamlines):
    streamlines, lengths = get_fake_streamlines
    arseq = RolloutEnvironment._padded_streamlines_to_array_sequence(
        streamlines, lengths
    )

    # Get reference array sequence
    ref_arseq = ArraySequence()
    for i in range(streamlines.shape[0]):
        s = streamlines[i, :lengths[i], :]
        ref_arseq.append(s)

    # Check that the two array sequences are equal
    assert len(arseq) == len(ref_arseq)
    for i in range(len(arseq)):
        assert np.array_equal(arseq[i], ref_arseq[i])
    
