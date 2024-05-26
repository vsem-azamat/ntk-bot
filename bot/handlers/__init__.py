from .public import router as public_router
from .data import router as data_router
from .anon import router as anon_router
from .admin import router as admin_router
from .gpt import router as gpt_router

from aiogram import Router

router = Router()

router.include_router(data_router)
router.include_router(public_router)
router.include_router(anon_router)
router.include_router(admin_router)
router.include_router(gpt_router) # Must be the last
