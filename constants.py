import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

DEMONSTRATIVE_PRONOUNS = {
    r"das",
    r"da(hinte[nr]|rüber)?",
    r"dies(e([mnrs])?)?",
}
DEMONSTRATIVE_PRONOUNS = {re.compile(pattern, re.IGNORECASE) for pattern in DEMONSTRATIVE_PRONOUNS}
