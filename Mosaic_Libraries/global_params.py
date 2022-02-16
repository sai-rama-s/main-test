import boto3
import os
import logging
import requests
from dataclasses import dataclass, asdict


@dataclass
class Client:
    url: str
    headers: dict

    def run_query(self, query: str, variables: dict, extract=False):
        request = requests.post(
            self.url,
            headers=self.headers,
            json={"query": query, "variables": variables},
        )
        assert request.ok, f"Failed with code {request.status_code}"
        return request.json()

    update_status = lambda self, orderId, status: self.run_query(
        """
            mutation MyMutation($_id: uuid!, $_status: String) {
					  update_order_details_by_pk(pk_columns: {id: $_id}, _set: {status: $_status}) {
				    id
				    status
				    instance_id
  }
}
        """,
        {"_id": orderId, "_status": status},
    )


HASURA_URL = "https://galaxeye-airborne.hasura.app/v1/graphql"
HASURA_HEADERS = {"X-Hasura-Admin-Secret": 'ex2IRh1w1b3ikgYBao8GuFHhsMmGKwm10p1M6wB2mFm86p44wQ0QVOjdmplKli2s'}

client = Client(url=HASURA_URL, headers=HASURA_HEADERS)


def query_handler(orderID, status):
    user_response = client.update_status(orderID, status)
    if user_response.get("errors"):
        return {"message": user_response["errors"][0]["message"]}, 400
    else:
        user = user_response["data"]["update_order_details_by_pk"]
        return user


# print(query_handler("0731a96a-4eaa-4dd4-bf6a-a5b39c693dc0", "Test-1"))


PARAMS = {
    "outputs": {
        "s3_bucket": 'airborne-data',
        "s3_path": "sentinel/orders/"
    }
}

# Connect to S3
s3_client = boto3.client('s3',
                         aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                         aws_secret_access_key='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'
                         )

ecs = boto3.client('ecs',
                    aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                     aws_secret_access_key='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'
                     )

# Upload to S3
logging.info("S3 connection successful")


if os.environ['ORDER_ID'] is not None:
    order_id = os.environ['ORDER_ID']
else:
    order_id = "trial"
    tmp = query_handler(order_id, "Error")
    instance_id = tmp['instance_id']
    logging.info("Order-id not found")
    s3_client.upload("./log.txt", PARAMS["outputs"]["s3_bucket"], PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
    ecs.stop_task(task=instance_id, cluster="", reason="Terminated due to error")
