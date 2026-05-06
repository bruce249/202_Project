import json

from main import MATERIALS, TEST_MATERIALS, TRAIN_MATERIALS


def handler(request):
    payload = {
        "materials": sorted(MATERIALS.keys()),
        "train_materials": TRAIN_MATERIALS,
        "test_materials": TEST_MATERIALS,
    }
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
