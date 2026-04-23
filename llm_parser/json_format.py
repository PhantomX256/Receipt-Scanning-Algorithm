from pydantic import BaseModel, Field
from typing import List, Optional

class Item(BaseModel):
    name: str = Field(description="The name of the item purchased")
    quantity: int = Field(default=1, description="The quantity of the item")
    price: float = Field(description="The total price of this item/row")

class Receipt(BaseModel):
    store_name: Optional[str] = Field(description="The name of the store or restaurant")
    date: Optional[str] = Field(description="The date of the purchase in YYYY-MM-DD format")
    items: List[Item] = Field(description="List of items purchased")
    subtotal: Optional[float] = Field(description="The subtotal before tax and tip")
    tax: Optional[float] = Field(description="The tax amount")
    total: float = Field(description="The final total amount paid")

