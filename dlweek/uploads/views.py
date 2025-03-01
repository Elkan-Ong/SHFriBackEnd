from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os


@csrf_exempt
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]

        # Save the uploaded image temporarily
        file_path = default_storage.save(f"uploads/{image.name}", ContentFile(image.read()))
        full_path = default_storage.path(file_path)  # Get absolute path

        try:
            # TODO: Pass `full_path` to your CNN model for prediction
            prediction = "Fake"  # Placeholder for CNN model output

            # Delete the file after processing
            if os.path.exists(full_path):
                os.remove(full_path)

            return JsonResponse({"message": "Image processed", "prediction": prediction})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "No image provided"}, status=400)