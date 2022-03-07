import configparser
import os
import socket
import sys
import time

_CONFIG_FILE = "/etc/radiation-benchmarks.conf"


class LogHelper:
    def __init__(
            self,
            benchmark_name: str,
            test_header: str,
            max_errors: int = 500,
            max_infos: int = 500,
            interval_print: int = 10,
            kill_after_double_error: bool = True
    ):
        self.__has_end = False
        self.__kill_after_double_error = kill_after_double_error
        self.__max_errors_per_iter = max_errors
        self.__max_infos_per_iter = max_infos
        # Used to print the log only for some iterations,
        # equal 1 means print every iteration
        self.__iter_interval_print = interval_print
        #  Used to log max_error_per_iter details each iteration
        self.__log_error_detail_count = 0
        self.__log_info_detail_count = 0
        # Saves the last amount of error found for a specific iteration
        self.__last_iter_errors = 0
        # Saves the last iteration index that had an error
        self.__last_iter_with_errors = 0
        self.__kernels_total_errors = 0
        self.__kernels_total_infos = 0
        self.__kernel_time_acc = 0.0
        self.__iteration_number = 0.0
        self.__it_time_start = 0.0
        self.__kernel_time = 0.0
        # log example: 2021_11_15_22_08_25_cuda_trip_half_lava_ECC_OFF_fernando.log
        date = time.localtime(time.time())
        date_fmt = time.strftime('%Y_%m_%d_%H_%M_%S', date)
        # Read the config file
        config = configparser.ConfigParser()
        config.read(_CONFIG_FILE)
        log_dir = config["DEFAULT"]["logdir"]
        ecc_info_file = config["DEFAULT"]["eccinfofile"]
        self.__signal_cmd = config["DEFAULT"]["signalcmd"]

        with open(ecc_info_file) as fp:
            # First line must contains a int representing the ECC config
            ecc = int(fp.readline().strip())
            ecc_config = "ON" if ecc == 1 else "OFF"
        hostname = socket.gethostname()

        self.__log_file_name = f"{log_dir}/{date_fmt}_{benchmark_name}_ECC_{ecc_config}_{hostname}.log"
        # Writing the header to the file
        self.__write_to_file(f"#HEADER {test_header}")
        self.__write_to_file(time.strftime(f"#BEGIN Y:%Y M:%m D:%d Time:%H:%M:%S", date))

    def __write_to_file(self, buffer: str):
        try:
            with open(self.__log_file_name, "a") as fp:
                fp.write(buffer + "\n")
                fp.flush()
        except (OSError, IOError) as e:
            print(f"[ERROR in log_string(char *)] Unable to open file {self.__log_file_name}",
                  file=sys.stderr)

    def __str__(self):
        return self.__log_file_name

    def __repr__(self):
        return str(self)

    def __del__(self):
        self.__end_log_file()

    def __end_log_file(self):
        if self.__has_end is False:
            self.__write_to_file("#END")
            self.__has_end = True

    def start_iteration(self):
        self.__log_error_detail_count = 0
        self.__log_info_detail_count = 0
        self.__it_time_start = time.time()

    def end_iteration(self):
        self.__kernel_time = time.time() - self.__it_time_start
        self.__kernel_time_acc += self.__kernel_time
        if self.__iteration_number % self.__iter_interval_print == 0:
            error_str = f"#IT Ite:{self.__iteration_number} "
            error_str += f"KerTime:{self.__kernel_time:.6f} "
            error_str += f"AccTime:{self.__kernel_time_acc:.6f}"
            self.__write_to_file(error_str)
        self.__iteration_number += 1

    def log_error_count(self, kernel_errors: int):
        self.__update_timestamp()
        if kernel_errors > 0:
            self.__kernels_total_errors += kernel_errors
            # (iteration_number-1) because this function is called after end_iteration()
            # that increments iteration_number

            error_str = f"#SDC Ite:Ite:{self.__iteration_number - 1} "
            error_str += f"KerTime:{self.__kernel_time:.6f} "
            error_str += f"AccTime:{self.__kernel_time_acc:.6f} "
            error_str += f"KerErr:{kernel_errors} AccErr:{self.__kernels_total_errors}",
            self.__write_to_file(error_str)
            if kernel_errors > self.__max_errors_per_iter:
                self.__write_to_file("#ABORT too many errors per iteration")
                self.__end_log_file()
                exit(-1)

            if (kernel_errors == self.__last_iter_errors
                    and (self.__last_iter_with_errors + 1) == self.__iteration_number
                    and self.__kill_after_double_error is True):
                self.__write_to_file("#ABORT amount of errors equals of the last iteration")
                self.__end_log_file()
                exit(1)

            self.__last_iter_errors = kernel_errors
            self.__last_iter_with_errors = self.__iteration_number

    def log_error_detail(self, error_detail: str):
        # TODO: NOT THREAD SAFE
        self.__log_error_detail_count += 1
        if self.__log_error_detail_count <= self.__max_errors_per_iter:
            self.__write_to_file(f"#ERR {error_detail}")

    def log_info_count(self, info_count: int):
        self.__update_timestamp()
        if info_count > 0:
            self.__kernels_total_infos += info_count
            # (iteration_number-1) because this function is called
            # after end_iteration() that increments iteration_number
            info_str = f"#CINF Ite:{self.__iteration_number - 1} KerTime:{self.__kernel_time:.6f} "
            info_str += f"AccTime:{self.__kernel_time_acc:.6f} KerInfo:{info_count} "
            info_str += f"AccInfo:{self.__kernels_total_infos}"
            self.__write_to_file(info_str)

    def log_info_detail(self, info_detail: str):
        self.__log_info_detail_count += 1
        if self.__log_info_detail_count <= self.__max_infos_per_iter:
            self.__write_to_file(f"#INF {info_detail}")

    def __update_timestamp(self):
        os.system(self.__signal_cmd)
        with open()
