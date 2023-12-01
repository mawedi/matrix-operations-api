from django.contrib import admin
from .models import (
    MatrixOperation,
    MatrixInverseTranspose,
    MatrixDeterminant,
    MatrixRank,
    Matrix
)

# Register your models here.
admin.site.register(MatrixOperation)
admin.site.register(MatrixInverseTranspose)
admin.site.register(MatrixRank),
admin.site.register(MatrixDeterminant)
admin.site.register(Matrix)