from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

session = WolframLanguageSession()
print(session.evaluate(wlexpr("Range[5]")))
