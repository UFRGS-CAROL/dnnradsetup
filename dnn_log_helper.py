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


def start_log_file(bench_name: str, header: str):
    if NOT_GOLDEN_GENERATION:
        log_helper.start_log_file(benchmark_name=bench_name, test_info=header)


def end_log_file():
    if NOT_GOLDEN_GENERATION:
        log_helper.end_log_file()


def disable_logging():
    global NOT_GOLDEN_GENERATION
    NOT_GOLDEN_GENERATION = False


def log_info_detail(info_detail: str):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_info_detail(info_detail)


def log_error_detail(error_detail: str):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_error_detail(error_detail)


def log_error_count(error_count: int):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_error_count(error_count)


def log_info_count(info_count: int):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_info_count(info_count)


def set_max_errors_iter(max_errors):
    if NOT_GOLDEN_GENERATION:
        return log_helper.set_max_errors_iter(max_errors)


def set_max_infos_iter(max_infos):
    if NOT_GOLDEN_GENERATION:
        return log_helper.set_max_infos_iter(max_infos)
