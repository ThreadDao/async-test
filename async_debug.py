from pymilvus import (
    DataType,
    MilvusClient,
    AsyncMilvusClient,
    AnnSearchRequest,
    RRFRanker,
)
import numpy as np
import asyncio
import time
import random
from util_log import test_log as log

#

num_entities, dim = 10000, 128
nq, default_limit = 2, 3
collection_name = "aaaa"
rng = np.random.default_rng(seed=19530)
output_fields = ["pk", "random"]

uri = "http://10.104.16.31:19530"
milvus_client = MilvusClient(uri=uri)


async def recreate_collection(_client):
    async_client = AsyncMilvusClient(uri)
    log.info("Start dropping all collection")
    for c in milvus_client.list_collections():
        await async_client.drop_collection(c)
    log.info("Dropping collection done")

    log.info("Start creating collection")
    await async_client.create_collection(collection_name, dimension=dim, consistency_level="Strong")
    log.info("Creating collection done")

if __name__ == '__main__':
    log.info(asyncio.get_running_loop()._thread_id)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(recreate_collection())