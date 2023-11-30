from django.db import models
from django.contrib.postgres.fields import ArrayField 
import uuid

# Create your models here.
class Matrix(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    matrix = ArrayField(ArrayField(models.FloatField()))


class MatrixDeterminant(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    matrix = ArrayField(ArrayField(models.FloatField()))
    determinant = models.FloatField()


class MatrixRank(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    matrix = ArrayField(ArrayField(models.FloatField()))
    rank = models.IntegerField()


class MatrixOperation(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    first_matrix = ArrayField(ArrayField(models.FloatField()))
    second_matrix = ArrayField(ArrayField(models.FloatField()))
    result = ArrayField(ArrayField(models.FloatField()))


class MatrixInverse(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    matrix = ArrayField(ArrayField(models.FloatField()))
    inverse = ArrayField(ArrayField(models.FloatField()))


class MatrixTranspose(models.Model):
    _id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    matrix = ArrayField(ArrayField(models.FloatField()))
    transpose = ArrayField(ArrayField(models.FloatField()))