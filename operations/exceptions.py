from django.utils.translation import gettext_lazy as _

from rest_framework.exceptions import APIException
from rest_framework import status

# Create your exceptions
class SingularMatrixException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("The matrix is singular and does not have an inverse.")
    default_code = "Error"


class DivisionByZeroException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("The matrix has 0 in the diagonal.")
    default_code = "Error"


class PositiveMatrixException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("The matrix is not positive.")
    default_code = "Error"


class ConvergenceMatrixException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("The matrix is divergent")
    default_code = "Error"


class FundamentalMinorsIncludeZero(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("Fundamental minor is 0.")
    default_code = "Error"