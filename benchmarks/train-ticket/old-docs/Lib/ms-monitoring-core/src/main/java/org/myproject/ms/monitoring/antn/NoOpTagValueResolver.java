

package org.myproject.ms.monitoring.antn;


class NoOpTagValueResolver implements TagValueResolver {
	@Override public String resolve(Object parameter) {
		return null;
	}
}
