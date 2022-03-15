"""
This wrapper is only if you don't want to use libLogHelper
"""

from libLogHelper.build import log_helper


def start_iteration():
    log_helper.start_iteration()


def end_iteration():
    log_helper.end_iteration()


def start_log_file(bench_name, header):
    log_helper.start_log_file(benchmark_name=bench_name, test_info=header)


def end_log_file():
    log_helper.end_log_file()

