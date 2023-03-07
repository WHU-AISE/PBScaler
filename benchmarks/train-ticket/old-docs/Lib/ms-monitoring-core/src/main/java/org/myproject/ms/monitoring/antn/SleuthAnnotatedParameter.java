
package org.myproject.ms.monitoring.antn;


class SleuthAnnotatedParameter {

	int parameterIndex;

	SpanTag annotation;

	Object argument;

	SleuthAnnotatedParameter(int parameterIndex, SpanTag annotation,
			Object argument) {
		this.parameterIndex = parameterIndex;
		this.annotation = annotation;
		this.argument = argument;
	}

}
