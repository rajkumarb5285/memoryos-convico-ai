import math

import numpy as np
import pytest

from app.embeddings.service import FastEmbedService


def test_embedding_dimension_and_finite():
    service = FastEmbedService()
    vec = service.embed_text("Test sentence for embedding validation.")

    assert len(vec) == 384
    assert all(not math.isnan(x) and not math.isinf(x) for x in vec)

    # Check approximate unit normalization
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.05


def test_embedding_batch_consistency():
    service = FastEmbedService()
    texts = ["I love programming in Python", "Indore is a beautiful clean city"]

    batch_vecs = service.embed_batch(texts)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384

    single_0 = service.embed_text(texts[0])
    single_1 = service.embed_text(texts[1])

    # Batch and single results should be identical
    np.testing.assert_allclose(batch_vecs[0], single_0, atol=1e-5)
    np.testing.assert_allclose(batch_vecs[1], single_1, atol=1e-5)


def test_nan_rejection():
    service = FastEmbedService()
    invalid_vec = [float("nan")] * 384

    with pytest.raises(ValueError, match="non-finite"):
        service._validate_vector(invalid_vec)
