from aiogram import Router

from .admin import router as admin_router
from .anon import router as anon_router
from .data import router as data_router
from .llm import router as llm_router
from .public import router as public_router

router = Router()

router.include_router(data_router)
router.include_router(public_router)
router.include_router(anon_router)
router.include_router(admin_router)
router.include_router(llm_router)  # Must be the last
