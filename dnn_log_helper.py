"""
This wrapper is only if you don't want to use libLogHelper
"""

from libLogHelper.build import log_helper

NOT_GOLDEN_GENERATION = True


def start_iteration():
    if NOT_GOLDEN_GENERATION:
        log_helper.start_iteration()


def end_iteration():
    if NOT_GOLDEN_GENERATION:
        log_helper.end_iteration()


def start_log_file(bench_name, header):
    if NOT_GOLDEN_GENERATION:
        log_helper.start_log_file(benchmark_name=bench_name, test_info=header)


def end_log_file():
    if NOT_GOLDEN_GENERATION:
        log_helper.end_log_file()


def disable_logging():
    global NOT_GOLDEN_GENERATION
    NOT_GOLDEN_GENERATION = False
