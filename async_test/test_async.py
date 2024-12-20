from async_wrapper import AsyncMilvusClientWrapper
import pytest
import asyncio
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
log = TestLog('async_test').log


async def aaa(x, y):
    log.info("start aaaa")
    await asyncio.sleep(0.5)
    return x + y


async def bbb(x, y):
    log.info("start bbb")
    res = aaa(x, y)
    return res, y


@pytest.mark.asyncio()
async def test_aa():
    res = await bbb(1, 2)
    log.info(res)


@pytest.mark.asyncio()
async def test_aaa():
    res = await bbb(4, 5)
    log.info(res)
    await res
