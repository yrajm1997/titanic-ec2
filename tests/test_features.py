
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from titanic_model.config.core import config
from titanic_model.processing.features import age_col_tfr


def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = age_col_tfr(
        variables=config.model_config.age_var,  # cabin
    )
    assert np.isnan(sample_input_data.loc[709,'Age'])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[709,'Age'] == 21