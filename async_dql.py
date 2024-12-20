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


num_entities, dim = 50000, 128
nq, default_limit = 2, 3
collection_name = "hello_milvus"
rng = np.random.default_rng(seed=19530)


# uri = "http://10.104.16.31:19530"
uri = "https://in01-dfe07085050a3c3.aws-us-west-2.vectordb-uat3.zillizcloud.com:19532"
token = "xxx"
milvus_client = MilvusClient(uri=uri, token=token)
async_milvus_client = AsyncMilvusClient(uri=uri, token=token)

loop = asyncio.get_event_loop()

schema = async_milvus_client.create_schema(auto_id=False,
                                           description="hello_milvus is the simplest demo to introduce the APIs")
schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field("random", DataType.FLOAT)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("embeddings2", DataType.FLOAT_VECTOR, dim=dim)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name="embeddings", index_type="HNSW", metric_type="L2", M=30, efConstruction=200)
index_params.add_index(field_name="embeddings2", index_type="IVF_SQ8", metric_type="L2", nlist=64)


async def recreate_collection():
    log.info("Start dropping collection")
    await async_milvus_client.drop_collection(collection_name)
    log.info("Dropping collection done")
    log.info("Start creating collection")
    await async_milvus_client.create_collection(collection_name, schema=schema, index_params=index_params,
                                                consistency_level="Strong")
    log.info("Creating collection done")


log.info((milvus_client.list_collections()))
has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    loop.run_until_complete(recreate_collection())
else:
    log.info("Start creating collection")
    loop.run_until_complete(
        async_milvus_client.create_collection(collection_name, schema=schema, index_params=index_params,
                                              consistency_level="Strong"))
    log.info("Creating collection done")

log.info(f"    all collections: {milvus_client.list_collections()}")
log.info(f"schema of collection {collection_name}")
log.info(milvus_client.describe_collection(collection_name))


async def async_insert(collection_name):
    rows = [
        {"pk": str(i), "random": float(i), "embeddings": [random.random() for _ in range(dim)],
         "embeddings2": [random.random() for _ in range(dim)]}
        for i in range(num_entities)]
    log.info(("Start async inserting entities"))

    start_time = time.time()
    tasks = []
    for i in range(0, num_entities, 1000):
        task = async_milvus_client.insert(collection_name, rows[i:i + 1000])
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    log.info("Total time: {:.2f} seconds".format(end_time - start_time))
    log.info("Async inserting entities done")
    return results


insert_res = loop.run_until_complete(async_insert(collection_name))
for r in insert_res:
    log.info(r['insert_count'])


async def other_async_task(collection_name):
    tasks = []
    output_fields = ["pk", "random"]
    # search
    log.info("search")
    for i in range(2000):
        random_vector = [[random.random() for _ in range(dim)] for _ in range(nq)]
        task = async_milvus_client.search(collection_name, random_vector, limit=default_limit, output_fields=output_fields,
                                          anns_field="embeddings")
        tasks.append(task)

    for i in range(2000):
        random_vector = [[random.random() for _ in range(dim)] for _ in range(nq)]
        task = async_milvus_client.search(collection_name, random_vector, limit=default_limit, output_fields=output_fields,
                                          anns_field="embeddings")
        tasks.append(task)

    # hybrid search
    search_param = {
        "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
        "anns_field": "embeddings",
        "param": {"metric_type": "L2"},
        "limit": default_limit,
        "expr": "random > 0.5"}
    req = AnnSearchRequest(**search_param)

    search_param2 = {
        "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
        "anns_field": "embeddings2",
        "param": {"metric_type": "L2"},
        "limit": default_limit
    }
    req2 = AnnSearchRequest(**search_param2)
    log.info("hybrid_search")
    task = async_milvus_client.hybrid_search(collection_name, [req, req2], RRFRanker(), default_limit, output_fields=output_fields)
    tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


results = loop.run_until_complete(other_async_task(collection_name))
for r in results:
    assert len(r) == nq
    assert len(r[0]) == default_limit
log.info(f"test done {len(results)}")
