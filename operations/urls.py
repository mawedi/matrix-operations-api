from django.urls import path
from .views import *

# Create your api
urlpatterns = [
    path('multiply/', MultiplyMatrixAPIView.as_view(), name='multiply-matrix'),
    path('add/', AddMatrixAPIView.as_view(), name='add-matrix'),
    path('substract/', SubstractMatrixAPIView.as_view(), name='substract-matrix'),
    
    path('solve/', SolveMatrixAPIView.as_view(), name='solve-matrix'),
    
    path('inverse/', InverseMatrixAIPView.as_view(), name='inverse-matrix'),
    path('determinant/', DeterminantMatrixAPIView.as_view(), name='determinant-matrix'),
    path('rank/', RankMatrixAPIView.as_view(), name='rang-matrix'),
    path('transpose/', TransposeMatrixAPIView.as_view(), name='transpose-matrix'),

    path('determinant/<str:_id>/', RetrieveMatrixDeterminantAPIView.as_view(), name='retrieve-determinant-matrix'),
    path('rank/<str:_id>/', RetrieveMatrixRankAPIView.as_view(), name='retrieve-rang-matrix'),
    path('multiply/<str:_id>/', RetrieveMatrixOperationAPIView.as_view(), name='retrieve-multiply-result'),
    path('add/<str:_id>/', RetrieveMatrixOperationAPIView.as_view(), name='add-multiply-result'),
    path('substract/<str:_id>/', RetrieveMatrixOperationAPIView.as_view(), name='substract-multiply-result'),
    path('solve/<str:_id>/', RetrieveMatrixOperationAPIView.as_view(), name='retrieve-solving-result'),
    path('inverse/<str:_id>/', RetrieveMatrixInverseTransposeAPIView.as_view(), name='retrive-matrix-inverse'),
    path('transpose/<str:_id>/', RetrieveMatrixInverseTransposeAPIView.as_view(), name='retrive-matrix-transpose')
]