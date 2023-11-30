from django.contrib import admin
from .models import (
    MatrixOperation,
    MatrixInverse,
    MatrixDeterminant,
    MatrixRank,
    Matrix,
    MatrixTranspose
)

# Register your models here.
admin.site.register(MatrixOperation)
admin.site.register(MatrixInverse)
admin.site.register(MatrixRank),
admin.site.register(MatrixDeterminant)
admin.site.register(Matrix)
admin.site.register(MatrixTranspose)