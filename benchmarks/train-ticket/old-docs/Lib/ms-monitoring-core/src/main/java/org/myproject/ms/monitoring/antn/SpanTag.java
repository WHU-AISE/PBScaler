

package org.myproject.ms.monitoring.antn;

import java.lang.annotation.ElementType;
import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.springframework.core.annotation.AliasFor;


@Retention(RetentionPolicy.RUNTIME)
@Inherited
@Target(value = { ElementType.PARAMETER })
public @interface SpanTag {

	
	@AliasFor("key")
	String value() default "";

	
	@AliasFor("value")
	String key() default "";

	
	String expression() default "";

	
	Class<? extends TagValueResolver> resolver() default NoOpTagValueResolver.class;

}
