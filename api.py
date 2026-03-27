from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from predict import get_price_range
from buyer_recommend import recommend

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SellerRequest(BaseModel):
    make: str
    model: str
    year: int
    mileage: int
    body_type: str
    drive_type: str
    zipcode: str
    engine: Optional[str] = ""
    trim: Optional[str] = ""


class BuyerRequest(BaseModel):
    budget: int
    body_type: Optional[str] = None
    drive_type: Optional[str] = None
    make: Optional[str] = None
    max_mileage: Optional[int] = None
    min_year: Optional[int] = None
    zipcode: Optional[str] = None
    top_n: int = 10


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/seller/price")
def seller_price(req: SellerRequest):
    low, mid, high = get_price_range(
        make=req.make,
        model=req.model,
        year=req.year,
        mileage=req.mileage,
        body_type=req.body_type,
        drive_type=req.drive_type,
        zipcode=req.zipcode,
        engine=req.engine,
        trim=req.trim,
    )
    return {
        "competitive_price": low,
        "fair_market_price": mid,
        "premium_price": high,
    }


@app.post("/api/buyer/recommend")
def buyer_recommend_api(req: BuyerRequest):
    results = recommend(
        budget=req.budget,
        body_type=req.body_type,
        drive_type=req.drive_type,
        make=req.make,
        max_mileage=req.max_mileage,
        min_year=req.min_year,
        zipcode=req.zipcode,
        top_n=req.top_n,
    )

    print(type(results))

    if hasattr(results, "to_dict"):
        return {"results": results.to_dict(orient="records")}

    if isinstance(results, list):
        return {"results": results}

    return {"results": []}