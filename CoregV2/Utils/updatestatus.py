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
  }
}
        """,
        { "_id": orderId, "_status": status},
    )

HASURA_URL = "https://galaxeye-airborne.hasura.app/v1/graphql"
HASURA_HEADERS = {"X-Hasura-Admin-Secret": "ex2IRh1w1b3ikgYBao8GuFHhsMmGKwm10p1M6wB2mFm86p44wQ0QVOjdmplKli2s"}

client = Client(url=HASURA_URL, headers=HASURA_HEADERS)

def query_handler(orderID, status):
    user_response = client.update_status(orderID, status)
    if user_response.get("errors"):
        return {"message": user_response["errors"][0]["message"]}, 400
    else:
        user = user_response["data"]["update_order_details_by_pk"]
        return CreateUserOutput(**user).to_json()
