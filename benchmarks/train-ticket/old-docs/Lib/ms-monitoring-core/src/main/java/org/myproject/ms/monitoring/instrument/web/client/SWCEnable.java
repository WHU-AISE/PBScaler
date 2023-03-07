package org.myproject.ms.monitoring.instrument.web.client;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;

import java.lang.annotation.*;


@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.TYPE, ElementType.METHOD})
@Documented
@ConditionalOnProperty(value = "spring.sleuth.web.client.enabled", matchIfMissing = true)
@interface SWCEnable {
}
