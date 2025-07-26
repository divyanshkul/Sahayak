from fastapi import APIRouter
from app.models.item import ItemCreate, ItemResponse
from app.models.response import MessageResponse

router = APIRouter()

@router.post("/", response_model=ItemResponse)
def create_item(item: ItemCreate):
    return ItemResponse(id=1, **item.dict())

@router.get("/{item_id}", response_model=ItemResponse)
def read_item(item_id: int):
    return ItemResponse(
        id=item_id,
        name=f"Item {item_id}",
        description="A sample item",
        price=29.99
    )