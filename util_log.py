import logging
import sys


class TestLog:
    def __init__(self, logger):
        self.logger = logger

        self.log = logging.getLogger(self.logger)
        self.log.setLevel(logging.DEBUG)

        try:
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s]: "
                                          "%(message)s (%(filename)s:%(lineno)s)")
            # [%(process)s] process NO.

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

        except Exception as e:
            print("Can not use error : %s" % (str(e)))


"""All modules share this unified log"""
test_log = TestLog('async_test').log
