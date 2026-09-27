PRODUCT_DEPARTMENT_MAP = {

    "Checking or savings account":
        "Banking Operations",

    "Credit card":
        "Credit Card Operations",

    "Credit reporting or other personal consumer reports":
        "Credit Reporting / Compliance",

    "Debt collection":
        "Collections Department",

    "Debt or credit management":
        "Credit Management",

    "Money transfer, virtual currency, or money service":
        "Payments / Money Transfer",

    "Mortgage":
        "Mortgage Operations",

    "Payday loan, title loan, personal loan, or advance loan":
        "Lending Operations",

    "Prepaid card":
        "Prepaid Card Operations",

    "Student loan":
        "Student Loan Operations",

    "Vehicle loan or lease":
        "Vehicle Finance"
}


def recommend_department(product):

    return PRODUCT_DEPARTMENT_MAP.get(
        product,
        "General Customer Support"
    )