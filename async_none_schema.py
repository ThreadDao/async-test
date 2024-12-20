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
collection_name = "hello_milvus"
rng = np.random.default_rng(seed=19530)
output_fields = ["id"]

# uri = "http://10.104.16.31:19530"
uri = "https://in01-dfe07085050a3c3.aws-us-west-2.vectordb-uat3.zillizcloud.com:19532"
token = "xxx"
milvus_client = MilvusClient(uri=uri, token=token)
async_milvus_client = AsyncMilvusClient(uri=uri, token=token)

loop = asyncio.get_event_loop()


async def recreate_collection():
    log.info("Start dropping all collection")
    for c in milvus_client.list_collections():
        await async_milvus_client.drop_collection(c)
    log.info("Dropping collection done")

    log.info("Start creating collection")
    await async_milvus_client.create_collection(collection_name, dimension=dim, consistency_level="Strong")
    log.info("Creating collection done")


loop.run_until_complete(recreate_collection())
log.info(f"    all collections: {milvus_client.list_collections()}")
log.info(f"schema of collection {collection_name}")
log.info(milvus_client.describe_collection(collection_name))


# insert
async def async_insert(collection_name):
    rows = [
        {"id": i, "vector": [random.random() for _ in range(dim)]}
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
milvus_client.flush(collection_name)


# concurrent
async def dql_async_task():
    tasks = []
    #  search
    random_vector = [[random.random() for _ in range(dim)] for _ in range(nq)]
    task = async_milvus_client.search(collection_name, random_vector, limit=default_limit, output_fields=output_fields)
    tasks.append(task)

    # hybrid search
    search_param = {
        "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
        "anns_field": "vector",
        "param": {"metric_type": "COSINE"},
        "limit": default_limit,
        "expr": "random > 0.5"}
    req = AnnSearchRequest(**search_param)

    search_param2 = {
        "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
        "anns_field": "vector",
        "param": {"metric_type": "COSINE"},
        "limit": default_limit
    }
    req2 = AnnSearchRequest(**search_param2)
    log.info("hybrid_search")
    task = async_milvus_client.hybrid_search(collection_name, [req, req2], RRFRanker(), default_limit, output_fields=output_fields)
    tasks.append(task)

    # query
    ids = [0, 10, 1000, 635, 2, 3]
    task = async_milvus_client.query(collection_name=collection_name, filter=f"id in {ids}", limit=default_limit,
                                     output_fields=output_fields)
    tasks.append(task)

    # get
    log.info("get ids after upsert")
    task = async_milvus_client.get(collection_name=collection_name, ids=ids[:1])
    tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


async def sync_dml_dql_tasks():
    # query before delete
    _ids = [0]
    res_before = await async_milvus_client.query(collection_name=collection_name, filter=f"id in {_ids}",
                                                 limit=default_limit,
                                                 output_fields=["embeddings"])
    log.info(f"query res before upsert: {res_before}")
    assert len(res_before) == 1

    # upsert
    log.info("upsert id[0]")
    vector = [random.random() for _ in range(dim)]
    log.info(f"expected vector: {vector[:10]}")
    upsert_res = await async_milvus_client.upsert(
        collection_name=collection_name,
        data=[{"id": 0, "vector": vector }],
    )
    log.info(f"upsert res: {upsert_res}")

    # query after upsert
    res_after_upsert = await async_milvus_client.query(collection_name=collection_name, filter=f"id in {_ids}",
                                                 limit=default_limit,
                                                 output_fields=["vector"])
    log.info(f"query res before upsert: {res_after_upsert}")
    assert len(res_after_upsert) == 1
    assert abs(res_after_upsert[0]["vector"][0] - vector[0]) < 0.000001

    # delete
    delete_res = await async_milvus_client.delete(collection_name=collection_name, ids=_ids)
    log.info(f"delete res: {delete_res}")

    # get after delete
    res_after = await async_milvus_client.get(collection_name=collection_name, ids=_ids,
                                              limit=default_limit,
                                              output_fields=output_fields)
    log.info(f"get res after delete: {res_after}")
    assert len(res_after) == 0

    # upsert
    vector = [random.random() for _ in range(dim)]
    log.info(f"expected vector: {vector[:10]}")
    upsert_res = await async_milvus_client.upsert(
        collection_name=collection_name,
        data=[{"id": num_entities, "vector": vector}],
    )
    log.info(f"upsert res: {upsert_res}")
    res_after_upsert = await async_milvus_client.query(collection_name=collection_name, output_fields=["count(*)"])
    log.info(f"get res after delete: {res_after_upsert}")


results = loop.run_until_complete(dql_async_task())
log.info(len(results))

loop.run_until_complete(sync_dml_dql_tasks())