"""
Data aggregation package for CoreTempAI.

This package provides functionality to aggregate data from simulation run directories
and create a unified cache for further processing.
"""

from CoreTempAI.data_aggregation.data_aggregator import DataAggregator
from CoreTempAI.data_aggregation.aggregator_base import DataAggregatorBase
from CoreTempAI.data_aggregation.default_aggregator import DefaultAggregator
from CoreTempAI.data_aggregation.custom_aggregator import CustomAggregator

# Version
__version__ = '0.1.0'

# Export public classes
__all__ = [
    'DataAggregator',
    'DataAggregatorBase',
    'DefaultAggregator',
    'CustomAggregator'
] 