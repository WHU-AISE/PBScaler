

package org.myproject.ms.monitoring.instrument.web.client;

import java.io.IOException;

import org.myproject.ms.monitoring.instrument.web.HSInject;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.instrument.web.HTKInject;
import org.myproject.ms.monitoring.util.ExceptionUtils;
import org.springframework.http.HttpRequest;
import org.springframework.http.client.ClientHttpRequestExecution;
import org.springframework.http.client.ClientHttpRequestInterceptor;
import org.springframework.http.client.ClientHttpResponse;


public class TRTInter extends ATHRInter
		implements ClientHttpRequestInterceptor {

	public TRTInter(Chainer tracer, HSInject spanInjector,
			HTKInject httpTraceKeysInjector) {
		super(tracer, spanInjector, httpTraceKeysInjector);
	}

	@Override
	public ClientHttpResponse intercept(HttpRequest request, byte[] body,
			ClientHttpRequestExecution execution) throws IOException {
		publishStartEvent(request);
		return response(request, body, execution);
	}

	private ClientHttpResponse response(HttpRequest request, byte[] body,
			ClientHttpRequestExecution execution) throws IOException {
		try {
			return new THResp(this, execution.execute(request, body));
		} catch (Exception e) {
			if (log.isDebugEnabled()) {
				log.debug("Exception occurred while trying to execute the request. Will close the span [" + currentSpan() + "]", e);
			}
			this.tracer.addTag(Item.SPAN_ERROR_TAG_NAME, ExceptionUtils.getExceptionMessage(e));
			finish();
			throw e;
		}
	}

}
