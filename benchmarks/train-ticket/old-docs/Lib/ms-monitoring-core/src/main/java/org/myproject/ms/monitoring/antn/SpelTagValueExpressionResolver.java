

package org.myproject.ms.monitoring.antn;

import java.lang.invoke.MethodHandles;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.expression.Expression;
import org.springframework.expression.ExpressionParser;
import org.springframework.expression.spel.standard.SpelExpressionParser;


class SpelTagValueExpressionResolver implements TagValueExpressionResolver {
	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	@Override
	public String resolve(String expression, Object parameter) {
		try {
			ExpressionParser expressionParser = new SpelExpressionParser();
			Expression expressionToEvaluate = expressionParser.parseExpression(expression);
			return expressionToEvaluate.getValue(parameter, String.class);
		} catch (Exception e) {
			log.error("Exception occurred while tying to evaluate the SPEL expression [" + expression + "]", e);
		}
		return parameter.toString();
	}
}
