import logging
import asyncio
from functools import wraps
from typing import Optional
from pymilvus import (
    AsyncMilvusClient,
    IndexType
)
from pymilvus.orm.types import CONSISTENCY_STRONG
from pymilvus.orm.collection import CollectionSchema

# 设置日志
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_execution(func):
    """装饰器，用于在异步函数执行前后记录日志"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        log.info(f"Start executing {func.__qualname__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)  # 执行异步函数
            log.info(f"Completed {func.__qualname__}, result: {result}")
            return result
        except Exception as e:
            log.error(f"Error in {func.__qualname__}: {e}", exc_info=True)
            raise

    return wrapper


class AsyncMilvusClientWrapper:
    async_milvus_client = None

    @log_execution
    def init_async_client(self, uri: str = "http://localhost:19530",
                          user: str = "",
                          password: str = "",
                          db_name: str = "",
                          token: str = "",
                          timeout: Optional[float] = None,
                          active_trace=False,
                          check_task=None, check_items=None,
                          **kwargs):
        self.active_trace = active_trace

        """ In order to distinguish the same name of collection """
        self.async_milvus_client = AsyncMilvusClient(uri, user, password, db_name, token, timeout, **kwargs)


    @log_execution
    async def create_collection(
            self,
            collection_name: str,
            dimension: Optional[int] = None,
            primary_field_name: str = "id",
            id_type: str = "int",
            vector_field_name: str = "vector",
            metric_type: str = "COSINE",
            auto_id: bool = False,
            timeout: Optional[float] = None,
            schema: Optional[CollectionSchema] = None,
            index_params=None,
            **kwargs
    ):
        kwargs["consistency_level"] = kwargs.get("consistency_level", CONSISTENCY_STRONG)

        # 使用 api_request 执行实际任务
        self.async_milvus_client.create_collection(collection_name, dimension, primary_field_name,
             id_type, vector_field_name, metric_type, auto_id, timeout, schema, index_params, **kwargs)
