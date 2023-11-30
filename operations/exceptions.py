from django.utils.translation import gettext_lazy as _

from rest_framework.exceptions import APIException
from rest_framework import status

# Create your exceptions
class SingularMatrixException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("The matrix is singular and does not have an inverse.")
    default_code = "Error"