

package org.myproject.ms.monitoring.antn;

import org.aopalliance.intercept.MethodInvocation;
import org.myproject.ms.monitoring.Item;


public interface SpanCreator {

	
	Item createSpan(MethodInvocation methodInvocation, NewSpan newSpan);
}
