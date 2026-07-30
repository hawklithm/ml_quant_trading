class QuantPipelineError(RuntimeError):
    """Base class for expected, reportable pipeline failures."""


class DataUnavailable(QuantPipelineError):
    """Required market or auxiliary data could not be loaded."""


class StaleData(QuantPipelineError):
    """Data exists but is outside the configured freshness window."""


class ModelTrainingError(QuantPipelineError):
    """A model could not be trained or scored for the requested ticker."""


class ExecutionRejected(QuantPipelineError):
    """An order failed validation and was not applied to paper state."""
