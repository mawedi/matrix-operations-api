from rest_framework import serializers

from .models import (
    Matrix,
    MatrixDeterminant,
    MatrixRank,
    MatrixOperation,
    MatrixInverseTranspose
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


class MatrixInverseTransposeSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatrixInverseTranspose
        fields = '__all__'
