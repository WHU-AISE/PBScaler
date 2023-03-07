package preserveOther.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import preserveOther.entity.OrderTicketsInfo;

/**
 * @author fdse
 */
public interface PreserveOtherService {

    Response preserve(OrderTicketsInfo oti, HttpHeaders headers);
}
