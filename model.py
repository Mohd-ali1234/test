from pydantic import BaseModel
from typing import List

class AppUsageItem(BaseModel):
    package_name: str
    usage_sec: int

class AppUsageBatch(BaseModel):
    user_id: str
    apps: List[AppUsageItem]
