from fastapi import APIRouter
from .FR_route import router as FR_cls_route


router = APIRouter()
router.include_router(FR_cls_route, prefix="/face_recognition")