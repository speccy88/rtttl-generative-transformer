"""Convenience exports; model implementations remain small and directly readable."""
from .transformer import TransformerLM, count_parameters
from .recurrent import GRULM
__all__ = ['TransformerLM', 'GRULM', 'count_parameters']
