from django.core.mail import send_mail
from django.shortcuts import render
from django.conf import settings
from django.template.loader import get_template

import requests

import json

from rest_framework.generics import (
    RetrieveAPIView
)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .exceptions import (
    SingularMatrixException,
    DivisionByZeroException
)
from .models import (
    Matrix,
    MatrixDeterminant,
    MatrixRank,
    MatrixOperation,
    MatrixInverseTranspose
)
from .serializers import (
    MatrixSerializer,
    MatrixDeterminantSerializer,
    NestedMatrixDeterminantSerializer,
    MatrixRankSerializer,
    NestedMatrixRankSerializer,
    MatrixOperationSerializer,
    MatrixInverseTransposeSerializer, 
)
from .utils import *
from .types import *
import numpy as np

# Create your views here.
class RetrieveMatrixOperationAPIView(RetrieveAPIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    queryset = MatrixOperation.objects.all()
    serializer_class = MatrixOperationSerializer
    lookup_field = '_id'


class RetrieveMatrixDeterminantAPIView(RetrieveAPIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    queryset = MatrixDeterminant.objects.all()
    serializer_class = NestedMatrixDeterminantSerializer
    lookup_field = '_id'


class RetrieveMatrixRankAPIView(RetrieveAPIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    queryset = MatrixRank.objects.all()
    serializer_class = NestedMatrixRankSerializer
    lookup_field = '_id'


class RetrieveMatrixInverseTransposeAPIView(RetrieveAPIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    queryset = MatrixInverseTranspose.objects.all()
    serializer_class = MatrixInverseTransposeSerializer
    lookup_field = '_id'


class MultiplyMatrixAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        # Getting data from the request
        first_matrix = request.data.get('first_matrix')
        second_matrix = request.data.get('second_matrix', None)
        first_matrix_type = request.data.get('first_matrix_type', None)
        second_matrix_type = request.data.get('second_matrix_type', None)
        m_first_matrix = int(request.data.get('m_first_matrix', 0))
        m_second_matrix = int(request.data.get('m_second_matrix', 0))

        if first_matrix_type is None or second_matrix_type is None:
            return Response({"message": "Types of matrix are missing!"}, status=status.HTTP_400_BAD_REQUEST)

        # Verification of the first matrix form
        serializer_instance_first_matrix = MatrixSerializer(data={'matrix': first_matrix})
        if not(serializer_instance_first_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)
        
        first_matrix = serializer_instance_first_matrix.validated_data.get('matrix')
            
        # Calculating the inverse of the matrix or the transpose considering the type provided
        result = None
        if second_matrix is None:
            if second_matrix_type == INVERSE:
                second_matrix = np.linalg.inv(first_matrix)
                result = multiply_banded_matrix_inverse(first_matrix, second_matrix, m_first_matrix)
                
            elif second_matrix_type == TRANSPOSE:
                second_matrix = transpose_banded_matrix(first_matrix)
                result = multiply_banded_matrix_transpose(first_matrix, m_first_matrix)
            
        else:
            serializer_instance_second_matrix = MatrixSerializer(data={'matrix': second_matrix})
            if not(serializer_instance_second_matrix.is_valid(raise_exception=True)):
                return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)
        
            second_matrix = serializer_instance_second_matrix.validated_data.get('matrix')
        
        # Calculating the matrix considering their types
        if result is None:
            if first_matrix_type == UPPER_MATRIX and second_matrix_type == LOWER_MATRIX:
                result = multiply_upper_lower_triangular(first_matrix, second_matrix)
            elif first_matrix_type == UPPER_MATRIX and second_matrix_type == DENSE_MATRIX:
                result = multiply_upper_triangular_dense(first_matrix, second_matrix)
            elif first_matrix_type == LOWER_MATRIX and second_matrix_type == DENSE_MATRIX:
                result = multiply_lower_triangular_dense(first_matrix, second_matrix)
            elif first_matrix_type == BANDED_MATRIX and second_matrix_type == LOWER_BANDED_MATRIX:
                result = multiply_banded_lower_banded_matrix(first_matrix, second_matrix, m_first_matrix)
            elif first_matrix_type == LOWER_BANDED_MATRIX and second_matrix_type == UPPER_BANDED_MATRIX:
                result = multiply_lower_banded_upper_banded_matrix(first_matrix, second_matrix, m_first_matrix, m_second_matrix)
                
            elif first_matrix_type == BANDED_MATRIX and second_matrix_type == DENSE_MATRIX:
                result = multiply_banded_dense(first_matrix, second_matrix, m_first_matrix)
            elif first_matrix_type == LOWER_BANDED_MATRIX and second_matrix_type == DENSE_MATRIX:
                result = multiply_lower_banded_dense(first_matrix, second_matrix, m_first_matrix)
            
            # Vector multiplication    
            elif first_matrix_type == DENSE_MATRIX and second_matrix_type == VECTOR:
                result = multiply_dense_vector(first_matrix, second_matrix)
            elif first_matrix_type == LOWER_MATRIX and second_matrix_type == VECTOR:
                result = multiply_lower_vector(first_matrix, second_matrix)
            elif first_matrix_type == UPPER_MATRIX and second_matrix_type == VECTOR:
                result = multiply_upper_vector(first_matrix, second_matrix)
            elif first_matrix_type == LOWER_BANDED_MATRIX and second_matrix_type == VECTOR:
                result = multiply_lower_banded_vector(first_matrix, second_matrix, m_first_matrix)
            elif first_matrix_type == UPPER_BANDED_MATRIX and second_matrix_type == VECTOR:
                result = multiply_upper_banded_vector(first_matrix, second_matrix, m_first_matrix)
            else:
                result = multiply_dense_dense(first_matrix, second_matrix)

        # Saving the result
        data = {
            'first_matrix': first_matrix,
            'second_matrix': second_matrix,
            'result': result
        }
        result_serializer_instance = MatrixOperationSerializer(data=data)
        if result_serializer_instance.is_valid(raise_exception=True):
            saved_instance = result_serializer_instance.save()

        return Response({'_id': saved_instance._id}, status=status.HTTP_201_CREATED)


class AddMatrixAPIView(APIView): 
    permission_classes = [AllowAny, ]
    authentication_classes = []
        
    def post(self, request, *args, **kwargs):
        # Getting the data from the request
        first_matrix = request.data.get('first_matrix')
        second_matrix = request.data.get('second_matrix')

        # Verification of the matrix form
        serializer_instance_first_matrix = MatrixSerializer(data={'matrix': first_matrix})
        if not(serializer_instance_first_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)

        serializer_instance_second_matrix = MatrixSerializer(data={'matrix': second_matrix})
        if not(serializer_instance_second_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Calculating the matrix 
        first_matrix = serializer_instance_first_matrix.validated_data.get('matrix')
        second_matrix = serializer_instance_second_matrix.validated_data.get('matrix')
        result = add_matrix_to_matrix(first_matrix, second_matrix)
        
        # Saving the result 
        data = {
            'first_matrix': first_matrix,
            'second_matrix': second_matrix,
            'result': result
        }
        result_serializer_instance = MatrixOperationSerializer(data=data)
        if result_serializer_instance.is_valid(raise_exception=True):
            saved_instance = result_serializer_instance.save()
        
        return Response({'_id': saved_instance._id}, status=status.HTTP_201_CREATED)


class SubstractMatrixAPIView(APIView): 
    permission_classes = [AllowAny, ]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        # Getting the data from the request
        first_matrix = request.data.get('first_matrix')
        second_matrix = request.data.get('second_matrix')


        # Verificatino of the matrix form
        serializer_instance_first_matrix = MatrixSerializer(data={'matrix': first_matrix})
        if not(serializer_instance_first_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)

        serializer_instance_second_matrix = MatrixSerializer(data={'matrix': second_matrix})
        if not(serializer_instance_second_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Calculation of the matrix
        first_matrix = serializer_instance_first_matrix.validated_data.get('matrix')
        second_matrix = serializer_instance_second_matrix.validated_data.get('matrix')
        result = substract_matrix_from_matrix(first_matrix, second_matrix)
        
        # Saving the result
        data = {
            'first_matrix': first_matrix,
            'second_matrix': second_matrix,
            'result': result
        }
        result_serializer_instance = MatrixOperationSerializer(data=data)
        if result_serializer_instance.is_valid(raise_exception=True):
            saved_instance = result_serializer_instance.save()
        
        return Response({'_id': saved_instance._id}, status=status.HTTP_201_CREATED)


class InverseMatrixAIPView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    
    def post(self, request, *args, **kwargs):
        # Getting the data from the request
        matrix = request.data.get('matrix')


        # Verificatino of the matrix form
        serializer_instance_matrix = MatrixSerializer(data={'matrix': matrix})
        if not (serializer_instance_matrix.is_valid(raise_exception=True)):
            return Response({"message": "Wrong Data!"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Calculation of the inverse of the matrix
        try:
            matrix = serializer_instance_matrix.validated_data.get('matrix')
            matrix_inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise SingularMatrixException({"message": "The matrix is singular and does not have an inverse."})
        
        # Saving the result
        data = {
            'matrix': matrix,
            'result': matrix_inverse
        }
        matrix_inverse_serializer = MatrixInverseTransposeSerializer(data=data)
        if matrix_inverse_serializer.is_valid(raise_exception=True):
            result_instance = matrix_inverse_serializer.save()

        return Response({'_id': result_instance._id}, status=status.HTTP_201_CREATED)


class DeterminantMatrixAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        # Getting the data from the request
        matrix = request.data.get('matrix')

        
        # Verification of the matrix form
        serializer_matrix_instance = MatrixSerializer(data={'matrix': matrix})
        if serializer_matrix_instance.is_valid(raise_exception=True):
            matrix_instance = serializer_matrix_instance.save()

        # Calculation of the determinant of the matrix
        try:
            matrix = serializer_matrix_instance.validated_data.get('matrix')
            determinant = np.linalg.det(matrix)
        
        except np.linalg.LinAlgError:
            raise SingularMatrixException("The matrix is not singular.")

        # Saving the result
        serializer_matrix_det_instance = MatrixDeterminantSerializer(data={'determinant': determinant, 'matrix': matrix_instance.matrix})
        if serializer_matrix_det_instance.is_valid(raise_exception=True):
            result_instance = serializer_matrix_det_instance.save()

        return Response({'_id': result_instance._id}, status=status.HTTP_200_OK)


class RankMatrixAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
 
    def post(self, request, *args, **kwargs): 
        # Getting the data from the request
        matrix = request.data.get('matrix')

        
        # Verification of the matrix form
        serializer_matrix_instance = MatrixSerializer(data={'matrix': matrix})
        serializer_matrix_instance.is_valid(raise_exception=True)

        # Calculation of the rank of the matrix
        matrix = serializer_matrix_instance.validated_data.get('matrix')        
        determinant = np.linalg.matrix_rank(matrix)
        
        # Saving the result
        serializer_matrix_rank_instance = MatrixRankSerializer(data={'rank': determinant, 'matrix': matrix})
        if serializer_matrix_rank_instance.is_valid(raise_exception=True):
            result_instance = serializer_matrix_rank_instance.save()
        
        return Response({'_id': result_instance._id}, status=status.HTTP_200_OK)

class SolveMatrixAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []
    
    def post(self, request, *args, **kwargs):
        # Getting the data from the request
        matrix = request.data.get('matrix')
        vector = request.data.get('vector')
        matrix_type = request.data.get('matrix_type', None)
        algorithm = request.data.get('algorithm', None)
        max_iteration = int(request.data.get('max_iteration', 0))
        m = int(request.data.get('m', 0))
        epsilon = float(request.data.get('epsilon', -1))
        
        
        # Verificatino of the matrix form
        serializer_matrix_instance = MatrixSerializer(data={'matrix': matrix})
        if not(serializer_matrix_instance.is_valid()): 
            return Response({"message": "Wrong data!"}, status=status.HTTP_400_BAD_REQUEST)

        serializer_vector_instance = MatrixSerializer(data={'matrix': vector})
        if not (serializer_vector_instance.is_valid()):
            return Response({"message": "Wrong data!"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Solving the matrix
        if matrix_type == UPPER_MATRIX:
            result = solve_upper_matrix(matrix, vector)
        elif matrix_type == LOWER_MATRIX:
            result = solve_lower_matrix(matrix, vector)
        elif matrix_type == UPPER_BANDED_MATRIX:
            result = solve_upper_banded_matrix(matrix, vector, m)
        elif matrix_type == LOWER_BANDED_MATRIX:
            result = solve_lower_banded_matrix(matrix, vector, m)

        elif matrix_type == DENSE_SYMMETRIC_MATRIX and algorithm == GAUSS_ELIMINATOR_SYMMETRIC_DENSE_MATRIX:
            result = solve_symmetric_desne_matrix_gauss_elimination(matrix, vector) 
        elif matrix_type == BANDED_SYMMETRIC_MATRIX and algorithm == GAUSS_ELIMINATOR_SYMMETRIC_BANDED_MATRIX:
            result = solve_symmetric_banded_matrix_gauss_elimination(matrix, vector, m)
        elif matrix_type == DENSE_SYMMETRIC_MATRIX and algorithm == GAUSS_JORDAN_SYMMETRIC_DENSE_MATRIX:
            result = solve_symmetric_matrix_gauss_jordan(matrix, vector)
        elif matrix_type == BANDED_SYMMETRIC_MATRIX and algorithm == GAUSS_JORDAN_SYMMETRIC_BANDED_MATRIX:
            result = solve_symmetric_matrix_gauss_jordan(matrix, vector) # False one
        elif matrix_type == DENSE_SYMMETRIC_MATRIX and algorithm == LU_DENSE_SYMMETRIC:
            result = solve_symmetric_dense_matrix_LU(matrix, vector)
        elif matrix_type == BANDED_SYMMETRIC_MATRIX and algorithm == LU_BANDED_SYMMETRIC:
            result = solve_symmetric_banded_matrix_LU(matrix, vector, m)
        
        elif matrix_type == DENSE_MATRIX and algorithm == CHOLESKY_DENSE_MATRIX:
            result = solve_cholesky_dense_matrix(matrix, vector)
        elif matrix_type == BANDED_MATRIX and algorithm == CHOLESKY_BANDED_MATRIX:
            result = solve_cholesky_banded_matrix(matrix, vector, m)

        elif matrix_type == DENSE_MATRIX and algorithm == PIVOT_PARTIEL_GAUSS_DENSE:
            result = solve_dense_matrix_pivot_partiel_gauss(matrix, vector)
        elif matrix_type == BANDED_MATRIX and algorithm == PIVOT_PARTIEL_GAUSS_BANDED:
            result = solve_dense_matrix_pivot_partiel_gauss(matrix, vector)
            
        elif algorithm == JACOBI:
            if epsilon < 0 and max_iteration > 1:
                result = solve_jacobi_with_max_iteration(matrix, vector, max_iteration)
            elif epsilon > 0 and max_iteration <= 0:
                result = solve_jacobi_with_epsilon(matrix, vector, epsilon)
            else:
                return Response({"message": "epsilon or max iteration must be provided."})
            
        elif algorithm == GAUSS_SEIDEL:
            if epsilon < 0 and max_iteration > 1:
                result = solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration)
            elif epsilon > 0 and max_iteration <= 0:
                result = solve_gauss_seidel_with_epsilon(matrix, vector, epsilon)
            else:
                return Response({"message": "epsilon or max iteration must be provided."})
        
        else:
            return Response({"message": "The informations are not provided correctly."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Saving the result
        data = {
            'first_matrix': matrix,
            'second_matrix': vector,
            'result': result
        }
        serializer_result_instance = MatrixOperationSerializer(data=data)
        if serializer_result_instance.is_valid(raise_exception=True):
            result_instance = serializer_result_instance.save()
    
        return Response({'_id': result_instance._id}, status=status.HTTP_201_CREATED)


class TransposeMatrixAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []

    def post(self, request, *args, **kwargs): 
        # Getting the data from the request
        matrix = request.data.get('matrix')


        serializer_matrix_instance = MatrixSerializer(data={'matrix': matrix})
        serializer_matrix_instance.is_valid(raise_exception=True)
        
        # Calculation of the transpose of the matrix
        matrix = serializer_matrix_instance.validated_data.get('matrix')
        transpose_matrix = np.transpose(matrix)
        
        # Saving the result
        serializer_result_instance = MatrixInverseTransposeSerializer(data={'matrix': matrix, 'result': transpose_matrix})
        if serializer_result_instance.is_valid(raise_exception=True):
            result_instance = serializer_result_instance.save()
        
        return Response({'_id': result_instance._id}, status=status.HTTP_201_CREATED) 


class SendEmailAPIView(APIView):
    permission_classes = [AllowAny, ]
    authentication_classes = []

    API_KEYS = [
        "https://emailvalidation.abstractapi.com/v1/?api_key=f699e39e0e4a41329cf106ffc8d4a0dc&email=",
        "https://emailvalidation.abstractapi.com/v1/?api_key=7dc65ad406214fc584fed69045cd35f0&email=",
        "https://emailvalidation.abstractapi.com/v1/?api_key=617b041537f74fd3bc05020743e7d73d&email=",
        "https://emailvalidation.abstractapi.com/v1/?api_key=7240d2707e0c48aab51259e4d19b78e2&email=",
    ]
    
    def verify_email(self, email):
        response = requests.get(f'{self.API_KEYS[0]}{email}')
        content = json.loads(response.content)
        if response.status_code == status.HTTP_200_OK:
            if content['deliverability'] == 'UNDELIVERABLE':
                return False
        
        elif response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pass

        elif response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
            self.API_KEYS.pop(0)
            self.verify_email(email)
            
        return True

    def post(self, request, *args, **kwargs):
        # Setting subject and message to send to user
        subject = "Vos précieux commentaires - Aidez-nous à améliorer votre expérience sur notre site Web"
        feedback = request.data.get('message', "")
        email_sender = request.data.get('email', "")
        first_name = request.data.get('fisrt_name', "")
        last_name = request.data.get('last_name', "")


        data_to_send = {
            "email": email_sender,
            "first_name": first_name,
            "last_name": last_name,
            "feedback": feedback
        }
        
        # Checking the email if exists
        try:
            checking_result = self.verify_email(email_sender)
            if checking_result is False:
                return Response({"message": "Vérifiez votre adresse e-mail!"}, status=status.HTTP_400_BAD_REQUEST)

        except IndexError:
            pass
        
        # Sending an email to the user who provides the feedback
        html_content = render(request, 'operations/sender.html', context=data_to_send).content.decode('utf-8')
        result_sending = send_mail(
            subject,
            feedback,
            settings.EMAIL_HOST_USER,
            [email_sender],
            html_message=html_content,
            fail_silently=False
        )

        # Sending the feedback to the group
        html_content = render(request, 'operations/receiver.html', context=data_to_send).content.decode('utf-8')
        result_sending = send_mail(
            subject,
            feedback,
            settings.EMAIL_HOST_USER,
            settings.EMAILS_RECEIVERS,
            html_message=html_content,
            fail_silently=False
        )
        
        return Response(status=status.HTTP_200_OK)
