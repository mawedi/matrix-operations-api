from rest_framework import serializers

from .models import (
    Matrix,
    MatrixDeterminant,
    MatrixRank,
    MatrixOperation,
    MatrixInverse,
    MatrixTranspose
)

# Create your serializers
class MatrixSerializer(serializers.ModelSerializer):
    class Meta:
        model = Matrix
        fields = '__all__'
    

class MatrixDeterminantSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixDeterminant
        fields = '__all__'


class MatrixOperationSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixOperation
        fields = '__all__'


class NestedMatrixDeterminantSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixDeterminant
        fields = '__all__'


class MatrixRankSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixRank
        fields = '__all__'


class NestedMatrixRankSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixRank
        fields = '__all__'


class MatrixInverseSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixInverse 
        fields = '__all__'


class MatrixTransposeSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixTranspose
        fields = '__all__'